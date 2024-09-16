# What is this project?

This is a [Cog](https://github.com/replicate/cog) project, created for matting images using a mask (could be binary or trimap), with the intent of refining the edges of the mask and make it more accurate.

This project gets [deployed on Replicate](https://replicate.com/bahaal-tech/image-matting) as a model in the `bahaal-tech` organisation.

# How to run?

## Setup
Clone the repo
```sh
git clone git@github.com:bahaal-tech/image-matting-cog.git
```

Set up the submodule. This project uses ViTMatte for image matting, which is included as a submodule.
```sh
git submodule init
```

Update the submodule
```sh
git submodule update
```

Populate `.env` reflecting variables mentioned in `.env.example`

## Running the cog

Create an `images` directory in the root directory of this project. Copy an image and its corresponding segmentation mask into it.

Run this command from the root directory of this project. To do this, you need `cog` CLI set up on your machine. Follow [these steps](https://replicate.com/bahaal-tech/image-matting/versions) to do it (no native support for Windows, hehe).
```sh
cog predict -i image=@images/image.jpg mask=@images/mask.png
```
Check other supported inputs in `predict.py`

# Building and Deployment

To build a Docker image for this cog, run
```sh
cog build -t image-matting
```

To deploy a model to Replicate, you need to be logged into your Replicate account in `cog` CLI. Run the following command to log in
```sh
cog login
```
Follow whatever the CLI prompts and you'll be logged in

Use the following command to deploy this model on Replicate
```sh
cog push r8.im/bahaal-tech/image-matting
```

After pushing the model to Replicate, you can check the versions [here](https://replicate.com/bahaal-tech/image-matting/versions).

## Firefighting

Sometimes, Replicate is down (more often than you'd think). If matting is blocking production work, here's what you can do

1. Spin up a server with A100 GPU
2. Set this project up on the server
3. Build a docker image for the model
    ```sh
    cog build -t image-matting
    ```
4. Run the following command to run this model and expose inference as an HTTP endpoint
    ```sh
    docker run -d -p 5000:5000 --gpus all image-matting
    ```
5. Refer [this commit on caimera-web](https://github.com/bahaal-tech/caimera-web/commit/0065b65b1fb2e6597d04828eb6c2c4795226ca3c) for changes on the app.