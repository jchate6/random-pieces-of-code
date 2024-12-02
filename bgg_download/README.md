Run in python envionrment.
add list of games to `input_list.csv` in same directory as `querry_pgg.py`.
`python3 query_bgg.py`
Pick game from list using number
"n" will give full list (up to 1000 titles)
Games that couldn't be matched will be placed in `bad_cache.txt`
Output results will be placed in `bgg_outputs.csv`

If the code trips, you can run it again, and it will not start over unless you clear the cache.
Do this with 
`python3 query_bgg.py --cc`

If the names aren't matching well, you can use "try hard" mode to remove anything after a "-" or "(" to create a broader search:
`python3 query_bgg.py --th`

You can add "-" to the name to try to make things easier to find.

After running once, pull out the results from `bgg_outputs.csv` and add them to the spreadsheet.
Then copy the names in `bad_cache.txt` into `input_list.csv` and modify as needed to search again.

and run `python3 query_bgg.py --th --cc`
