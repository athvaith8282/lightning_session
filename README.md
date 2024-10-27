# Lightning_session4
Dog Breed classification module using lightning hydra

## build 

To build docker file , run 

```bash
docker compose build
```

## Training 

To train the model, run the following command:

```bash
docker-compose up train
```


## Evaluation

To evaluate the model, run the following command:

```bash
docker-compose up evaluate
```

The model_path will be load from .env file from dogbreed volume

## Inference

To run inference, run the following command:

```bash
docker-compose up infer
```
The model_path will be load from .env file from dogbreed volume