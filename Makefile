# setup:
# 		python -m venv .env

# ifdef OS
# 	env_activate_command = .\.env\Scripts\activate
# else
# 	env_activate_command = source .env/bin/activate
# endif

# install:
# 		ifdef OS
# 			env_activate_command = .\.env\Scripts\activate
# 		else
# 			env_activate_command = source .env/bin/activate
# 		endif
# 		$(env_activate_command)

install:
	pip install --upgrade pip &&\
		pip install -r pip_reqs.txt

format:
	black .

test:
	python -m pytest --nbval notebook/03_data_exploration.ipynb


all: install format test