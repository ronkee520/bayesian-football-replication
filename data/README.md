# Data

`raw/epl_2023_24.csv` contains all 380 matches in the 2023/24 English Premier
League season. Each row is one match and the columns are:

| Column | Meaning |
| --- | --- |
| `home_team` | Home team name |
| `away_team` | Away team name |
| `home_goals` | Home goals scored |
| `away_goals` | Away goals scored |

The CSV is the fixed course dataset used for this replication. SHA-256:

`1b405d19e804f5221d65472ffc3fa0ca9044ba957c9a83775ca41bb47dab2dc0`

The source file has no dates or round labels. Rows appear in chronological match
order. The holdout experiment therefore treats the last 10 matches as one round
and documents this assumption in its output metadata.

The model uses goals and match outcomes only. League tables produced from these
rows report on-field points before administrative deductions and therefore differ
from the official final table for Everton and Nottingham Forest.

`processed/` contains generated deterministic derivatives. The source CSV remains
unchanged during the workflow.
