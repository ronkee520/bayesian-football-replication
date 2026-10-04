# Data

`raw/epl_2023_24.csv` contains all 380 matches in the 2023/24 English Premier
League season. Each row is one match and the columns are:

| Column | Meaning |
| --- | --- |
| `home_team` | Home team name |
| `away_team` | Away team name |
| `home_goals` | Home goals scored |
| `away_goals` | Away goals scored |

The file was supplied for the original course exercise and recovered byte for
byte from the legacy project history. SHA-256:

`1b405d19e804f5221d65472ffc3fa0ca9044ba957c9a83775ca41bb47dab2dc0`

The source file has no dates or round labels. Rows appear in chronological match
order. The holdout experiment therefore treats the last 10 matches as one round
and documents this assumption in its output metadata.

The model uses goals and match outcomes only. League tables produced from these
rows report on-field points before administrative deductions. Do not interpret
them as the official final Premier League table without separately applying the
Everton and Nottingham Forest deductions.

`processed/` is reserved for generated, deterministic derivatives. Raw data
must never be overwritten.
