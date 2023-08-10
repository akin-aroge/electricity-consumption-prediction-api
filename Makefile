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

docker_build:
	docker build -f Dockerfile -t "elec-demand-pred" .

docker_run:
	docker-compose  -f .\docker-compose.yml up -d

docker_it:
	docker run -it --rm elec-demand-pred  /bin/sh

all: install format test