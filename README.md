# FastEval Parkinsonism
![cover_image](./imgs/cover.png)
## Fast Installation at local by docker-compose
```bash
git clone <github-link>
# e.g. git clone git@github.com:CMDM-Lab/fasteval_parkinsonism.git
docker-compose up --build # for building
```

## Start
```bash
docker-compose up # start server or reactivate server
```

## Show
Open your browser and use the `https://localhost:13006`

## Stop
```bash
docker-compose down # stop server
# or use `ctrl+C`
```

## look at the linux environment
```bash
docker compose run web /bin/bash
```

## Hand Predictor API
Read the document for indenpendently usage ([Hand Predictor API](./src/lib/hand_predictor)).

## Website development  
Read the document for indenpendently usage ([Website development](./src)).  


## Other informations
1. More information about alternative installation in [TroubleShooting](./TroubleShooting.md).  
2. If you face a problem with message `A server is already running. Check /home/myuser/local/app/tmp/pids/server.pid.`, please look at the linux environment and remove the file `server.pid` (shown as the codes below).  
```bash
docker compose run web /bin/bash                # run the docker compose environment
rm /home/myuser/local/app/tmp/pids/server.pid   # remove the file
```
