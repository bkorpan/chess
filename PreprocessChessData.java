import java.io.*;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

        Pattern eventPattern = Pattern.compile("\\[Event \"Rated Classical game\"\\]");
        Pattern whitePattern = Pattern.compile("\\[White \"([^\"]+)\"\\]");
        Pattern blackPattern = Pattern.compile("\\[Black \"([^\"]+)\"\\]");
        Pattern variantPattern = Pattern.compile("\\[Variant \"([^\"]+)\"\\]");
        Pattern resultPattern = Pattern.compile("(1-0|0-1|1/2-1/2)$");

        int linesRead = 0;
        int resultsMatched = 0;
        int classicalGamesFound = 0;
        int gamesCounted = 0;

        // First pass: Count moves for each player
        try (BufferedReader reader = Files.newBufferedReader(inputPath)) {
            String line;
            boolean isRatedClassicalGame = false;
            String whitePlayer = null;
            String blackPlayer = null;

            while ((line = reader.readLine()) != null) {
                linesRead++;
                if (eventPattern.matcher(line).find()) {
                    isRatedClassicalGame = true;
                    classicalGamesFound++;
                }

                if (variantPattern.matcher(line).find()) {
                    isRatedClassicalGame = false;
                }

                Matcher whiteMatcher = whitePattern.matcher(line);
                if (whiteMatcher.find()) {
                    whitePlayer = whiteMatcher.group(1);
                }

                Matcher blackMatcher = blackPattern.matcher(line);
                if (blackMatcher.find()) {
                    blackPlayer = blackMatcher.group(1);
                }

                if (resultPattern.matcher(line).find()) {
                    resultsMatched++;
                    if (isRatedClassicalGame && whitePlayer != null && blackPlayer != null) {
                        gamesCounted++;
                        final int moveCount = (int) line.chars().filter(ch -> ch == '.').count();
                        playerMoveCount.put(whitePlayer, playerMoveCount.getOrDefault(whitePlayer, 0) + moveCount);
                        playerMoveCount.put(blackPlayer, playerMoveCount.getOrDefault(blackPlayer, 0) + moveCount);
                    }

                    isRatedClassicalGame = false;
                    whitePlayer = null;
                    blackPlayer = null;
                }
            }
        }
        System.out.println(linesRead);
        System.out.println(resultsMatched);
        System.out.println(classicalGamesFound);
        System.out.println(gamesCounted);

        int results_matched = 0;
        int games_written = 0;

        // Second pass: Write filtered games to the output file
        try (BufferedReader reader = Files.newBufferedReader(inputPath);
             BufferedWriter writer = Files.newBufferedWriter(outputPath)) {

            String line;
            StringBuilder gameStringBuilder = new StringBuilder();
            boolean isRatedClassicalGame = false;
            String whitePlayer = null;
            String blackPlayer = null;

            while ((line = reader.readLine()) != null) {
                //gameStringBuilder.append(line).append('\n');
                if (eventPattern.matcher(line).find()) {
                    isRatedClassicalGame = true;
                }

                if (variantPattern.matcher(line).find()) {
                    isRatedClassicalGame = false;
                }

                Matcher whiteMatcher = whitePattern.matcher(line);
                if (whiteMatcher.find()) {
                    gameStringBuilder.append(line).append('\n');
                    whitePlayer = whiteMatcher.group(1);
                }

                Matcher blackMatcher = blackPattern.matcher(line);
                if (blackMatcher.find()) {
                    gameStringBuilder.append(line).append('\n');
                    blackPlayer = blackMatcher.group(1);
                }

                if (resultPattern.matcher(line).find()) {
                    gameStringBuilder.append(line).append("\n\n");
                    results_matched++;
                    if (isRatedClassicalGame && whitePlayer != null && blackPlayer != null &&
                            (playerMoveCount.getOrDefault(whitePlayer, 0) >= minMoves ||
                                    playerMoveCount.getOrDefault(blackPlayer, 0) >= minMoves)) {
                        games_written++;
                        writer.write(gameStringBuilder.toString());
                    }

                    gameStringBuilder.setLength(0);
                    isRatedClassicalGame = false;
                    whitePlayer = null;
                    blackPlayer = null;
                }
            }
        }
        System.out.println(results_matched);
        System.out.println(games_written);
    }
}

