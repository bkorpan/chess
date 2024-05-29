# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import pickle
import time
import logging
import requests
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import mctx
import optax
import pgx
import boto3
from botocore.exceptions import ClientError
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from model import EncoderStack, Chessformer

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "chess"
    seed: int = 0
    max_num_iters: int = 200
    # network params
    model_size: int = 256
    num_layers: int = 6
    attn_size: int = 32
    num_heads: int = 8
    widening_factor: int = 1.5
    # selfplay params
    selfplay_batch_size: int = 64
    num_simulations: int = 8
    max_num_steps: int = 1024
    # training params
    training_batch_size: int = 128
    learning_rate: float = 3e-4
    # eval params
    eval_interval: int = 5

    ignore_checkpoint: bool = False
    mixed_precision: bool = True

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)


def forward_fn(x):
    encoder_stack = EncoderStack(
        num_heads = config.num_heads,
        num_layers = config.num_layers,
        attn_size = config.attn_size,
        widening_factor = config.widening_factor
    )
    net = Chessformer(
        encoder_stack = encoder_stack,
        model_size = config.model_size,
        num_tokens = 64
    )
    policy_out, value_out = net(x)
    return policy_out, value_out


forward = hk.transform_with_state(forward_fn)
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, None, state.observation)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, None, state.observation
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample, rng_key):
    (logits, value), model_state = forward.apply(
        model_params, model_state, rng_key, samples.obs
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample, rng_key):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data, rng_key
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key, my_model):
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0
    my_model_parmas, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        (my_logits, _), _ = forward.apply(
            my_model_parmas, my_model_state, None, state.observation
        )
        opp_logits = jnp.zeros(state.legal_action_mask.shape)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


def save_checkpoint(state, bucket_name, key):
    s3 = boto3.resource('s3')

    pickled_state = pickle.dumps(state)

    bucket = s3.Bucket(bucket_name)
    bucket.put_object(Key=key, Body=pickled_state)


def load_checkpoint(bucket_name, key):
    s3 = boto3.resource('s3')

    try:
        # Attempt to retrieve the object metadata
        obj = s3.Object(bucket_name, key)
        response = obj.get()
    except ClientError as e:
        # If a ClientError is raised, the object does not exist or you have no permission
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        else:
            # Handle other possible exceptions (e.g., permission issues, etc.)
            raise

    pickled_state = response['Body'].read()
    state = pickle.loads(pickled_state)

    return state


def check_for_interruption():
    try:
        # URL for the spot instance interruption notice metadata
        url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            # If there is a termination notice
            data = response.json()
            print("Interruption notice detected:")
            print(f"Action: {data['action']}")
            print(f"Time: {data['time']}")
            return True
        else:
            # No termination notice
            return False
    except requests.exceptions.RequestException as e:
        print("Failed to fetch interruption notice:", e)
        return False


def delete_object(bucket_name, key):
    # Create an S3 client
    s3 = boto3.client('s3')
    try:
        # Delete the object
        response = s3.delete_object(Bucket=bucket_name, Key=key)
        return response
    except Exception as e:
        print(f"Error occurred: {e}")


def terminate_spot_request_and_this_instance():
    def get_instance_metadata():
        """Retrieve instance metadata."""
        metadata_url = 'http://169.254.169.254/latest/meta-data/'
        instance_id = requests.get(metadata_url + 'instance-id').text
        region = requests.get(metadata_url + 'placement/availability-zone').text[:-1]
        return instance_id, region

    def get_spot_instance_request_id(instance_id, region):
        """Retrieve the Spot Instance Request ID for the given instance."""
        ec2_client = boto3.client('ec2', region_name=region)
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        spot_instance_request_id = response['Reservations'][0]['Instances'][0]['SpotInstanceRequestId']
        return spot_instance_request_id

    def cancel_spot_instance_request(spot_instance_request_id, region):
        """Cancel the Spot Instance Request."""
        ec2_client = boto3.client('ec2', region_name=region)
        ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_instance_request_id])

    def terminate_instance(instance_id, region):
        """Terminate the EC2 instance."""
        ec2_client = boto3.client('ec2', region_name=region)
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    instance_id, region = get_instance_metadata()
    spot_instance_request_id = get_spot_instance_request_id(instance_id, region)
    cancel_spot_instance_request(spot_instance_request_id, region)
    terminate_instance(instance_id, region)
    print(f"Spot Instance Request {spot_instance_request_id} canceled and instance {instance_id} terminated.")


def count_params(params):
    return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size, params)))


if __name__ == "__main__":
    # Configure mixed precision
    if config.mixed_precision:
        hk.mixed_precision.set_policy(hk.Linear, jmp.get_policy("params=float32,compute=bfloat16,output=float32"))
        hk.mixed_precision.set_policy(hk.MultiHeadAttention, jmp.get_policy("params=float32,compute=bfloat16,output=float32"))

    # s3 bucket for checkpointing
    bucket_name = "bkorpan-models"
    checkpoint_key = "checkpoint"

    if config.ignore_checkpoint:
        delete_object(bucket_name, checkpoint_key)

    # Load existing state if available
    state = load_checkpoint(bucket_name, checkpoint_key)
    if state is None:
        # Initialize model and opt_state
        dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
        dummy_input = dummy_state.observation
        model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
        #print(hk.experimental.tabulate(forward)(dummy_input))
        opt_state = optimizer.init(params=model[0])

        rng_key = jax.random.PRNGKey(config.seed)

        state = {
            "rng_key": rng_key,
            "model": model,
            "opt_state": opt_state,
            'iteration': 0,
            'hours': 0,
            'frames': 0
        }
    else:
        print("checkpoint loaded!")

    rng_key = state['rng_key']
    model = state['model']
    opt_state = state['opt_state']

    num_params = count_params(model)
    print(f"# parameters = {num_params}")

    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Initialize logging dict
    iteration = state['iteration']
    hours = state['hours']
    frames = state['frames']
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename=f'log.txt',
        filemode='w'
    )

    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model)
            log.update(
                {
                    f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

        if check_for_interruption():
            # Store checkpoint
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            state = {
                "rng_key": rng_key,
                "model": jax.device_get(model_0),
                "opt_state": jax.device_get(opt_state_0),
                "iteration": iteration,
                "frames": frames,
                "hours": hours,
            }
            save_checkpoint(state, bucket_name, checkpoint_key)
            exit()

        print(log)
        logging.info(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch, keys)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )

    # Store checkpoint of final state
    final_checkpoint_key = "model"
    model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
    state = {
        "rng_key": rng_key,
        "model": jax.device_get(model_0),
        "opt_state": jax.device_get(opt_state_0),
        "iteration": iteration,
        "frames": frames,
        "hours": hours,
    }
    save_checkpoint(state, bucket_name, final_checkpoint_key)

    # Delete old checkpoint, terminate spot request and the current instance
    #delete_object(bucket_name, checkpoint_key)
    terminate_spot_request_and_this_instance()
