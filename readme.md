# Detectron2 Sample Deployment
## Why this
Deploying a detectron based RNN model is difficult. In spite of the offical guide of deployment, it's hard to compile the environment for caffe2 or stuff like that. I know little about C++ and Linux, and most of the ML frameworks are not supporting Windows.
Anyway, My app is simple and I'm not worrying about the performance. I just want it works, and I'm saving my time.
## What's this
This is an interface for other programs to get the predict data of the detectron2 based RNN detecter.
## usage
The model file should be put in `engine/output/model_final.pth`, which is not included here.

Install Docker. 
```shell
 $ git clone https://github.com/CoccaGuo/Detectron2-Simple-Deployment.git
 $ cd Detectron2-Simple-Deployment
 ```

```shell
$ docker build -t cocca/co_detect
```

```shell
$ docker run -d -p 5000:5000 cocca/co_detect
```

Open your browser and goto `localhost:5000`.

## What's more
This project may only works with my model. Act with mildly caution.
