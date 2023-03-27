import chesspresso.game.Game;
import chesspresso.pgn.PGN;
import chesspresso.pgn.PGNReader;
import chesspresso.pgn.PGNWriter;
import chesspresso.pgn.PGNSyntaxError;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class PreprocessChessData {

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            System.err.println("Usage: java PreprocessChessData <input_pgn_file> <output_pgn_file> <min_moves>");
            System.exit(1);
        }

        Path inputPath = Paths.get(args[0]);
        Path outputPath = Paths.get(args[1]);
        int minMoves = Integer.parseInt(args[2]);

        Map<String, Integer> playerMoveCount = new HashMap<>();

        try (FileReader fileReader = new FileReader(inputPath.toFile())) {
            PGNReader pgnReader = new PGNReader(fileReader, "filename");

            Game game;
            while (true) {
                try {
                    game = pgnReader.parseGame();
                } catch (PGNSyntaxError e) {
                    System.err.println("Error parsing game: " + e.getMessage());
                    continue;
                }

                if (game == null) {
                    break;
                }

                String[] tags = game.getTags();
                boolean isRatedClassical = false;
                boolean isStandardVariant = true;

                for (String tag : tags) {
                    if (tag.startsWith("[Event \"Rated Classical game\"]")) {
                        isRatedClassical = true;
                    } else if (tag.startsWith("[Variant")) {
                        isStandardVariant = false;
                    }
                }

                if (isRatedClassical && isStandardVariant) {
                    String whitePlayer = game.getWhite();
                    String blackPlayer = game.getBlack();

                    playerMoveCount.put(whitePlayer, playerMoveCount.getOrDefault(whitePlayer, 0) + game.getMainLine().length / 2);
                    playerMoveCount.put(blackPlayer, playerMoveCount.getOrDefault(blackPlayer, 0) + game.getMainLine().length / 2);
                }
            }
        }

        try (FileReader fileReader = new FileReader(inputPath.toFile())) {
            PGNReader pgnReader = new PGNReader(fileReader, "filename");

            try (FileWriter fileWriter = new FileWriter(outputPath.toFile())) {
                PGNWriter pgnWriter = new PGNWriter(fileWriter);

                Game game;
                while (true) {
										try {
												game = pgnReader.parseGame();
										} catch (PGNSyntaxError e) {
												System.err.println("Error parsing game: " + e.getMessage());
												continue;
										}

										if (game == null) {
												break;
										}

                  	String[] tags = game.getTags();
                    boolean isRatedClassical = false;
                    boolean isStandardVariant = true;

                    for (String tag : tags) {
                        if (tag.startsWith("[Event \"Rated Classical game\"]")) {
                            isRatedClassical = true;
                        } else if (tag.startsWith("[Variant")) {
                            isStandardVariant = false;
                        }
                    }

                    String whitePlayer = game.getWhite();
                    String blackPlayer = game.getBlack();

                    if (isRatedClassical && isStandardVariant &&
                            playerMoveCount.get(whitePlayer) >= minMoves && playerMoveCount.get(blackPlayer) >= minMoves) {
                        pgnWriter.write(game.getModel());
                    }
                }
            }
        }
    }
}
