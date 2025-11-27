 # SDA25_Group11

data/		raw data<br/>
results/	figures and tables<br/>
code/		python code used to produce the figures/models


## Git Workflow

##### Starting on a new feature?
1. Update main
	- `git checkout main`: go to main branch
	- `git pull origin main`: update the main branch locally
2. Create your feature branch
	- `git checkout -b yourname/task-name`: this create a new local branch
3. Do your work + commit
	- `git add .` or `git add yourfile`
	- `git commit -m "message`
4. Before pushing, sync with main
	- `git pull --rebase origin main`: set upstream to remote branch and update
5. Push your branch
	- `git push -u origin yourname/task-name`
6. Open Pull Request on Github, get review and merge to main
##### Daily Start Routine
1. Make sure you are on your feature branch
	- `git checkout yourname/task-name`
2. Update local main first (always!)
	- `git checkout main`: go to local main
	- `git pull origin main`: update with remote main
3. Sync your local branch with latest main
	- `git checkout yourname/task-name`: go to your local branch
	- `git pull --rebase origin main`: update with main
4. Continue coding and push to main as previous