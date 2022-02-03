import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable

from nst import utils


@dataclass
class Config:
    content_img_name: str
    style_img_name: str
    height: int = 400
    content_weight: int = 1e5
    style_weight: int = 3e4
    tv_weight: int = 1
    optimizer: str = "lbfgs"
    iterations: int = 300
    model: str = "vgg19"
    init_method: str = "content"
    saving_freq: str = -1
    content_images_dir: str = "data/content-images"
    style_images_dir: str = "data/style-images"
    output_img_dir: str = "data/output-images"
    # num digits, ext
    img_format: str = (4, ".jpg")


def build_loss(
    neural_net,
    optimizing_img,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[
        content_feature_maps_index
    ].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction="mean")(
        target_content_representation, current_content_representation
    )

    style_loss = 0.0
    current_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(current_set_of_feature_maps)
        if cnt in style_feature_maps_indices
    ]
    for gram_gt, gram_hat in zip(
        target_style_representation, current_style_representation
    ):
        style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = (
        config.content_weight * content_loss
        + config.style_weight * style_loss
        + config.tv_weight * tv_loss
    )

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(
    neural_net,
    optimizer,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net,
            optimizing_img,
            target_representations,
            content_feature_maps_index,
            style_feature_maps_indices,
            config,
        )
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def transfer_style(config: Config) -> str:
    content_img_path = os.path.join(config.content_images_dir, config.content_img_name)
    style_img_path = os.path.join(config.style_images_dir, config.style_img_name)

    out_dir_name = (
        "combined_"
        + os.path.split(content_img_path)[1].split(".")[0]
        + "_"
        + os.path.split(style_img_path)[1].split(".")[0]
    )
    dump_path = os.path.join(config.output_img_dir, out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config.height, device)
    style_img = utils.prepare_img(style_img_path, config.height, device)

    if config.init_method == "random":
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(
            loc=0, scale=90.0, size=content_img.shape
        ).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config.init_method == "content":
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(
            style_img_path, np.asarray(content_img.shape[2:]), device
        )
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    (
        neural_net,
        content_feature_maps_index_name,
        style_feature_maps_indices_names,
    ) = utils.prepare_model(config.model, device)
    print(f"Using {config.model} in the optimization procedure.")

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[
        content_feature_maps_index_name[0]
    ].squeeze(axis=0)
    target_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(style_img_set_of_feature_maps)
        if cnt in style_feature_maps_indices_names[0]
    ]
    target_representations = [
        target_content_representation,
        target_style_representation,
    ]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = config.iterations

    #
    # Start of optimization procedure
    #
    if config.optimizer == "adam":
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(
            neural_net,
            optimizer,
            target_representations,
            content_feature_maps_index_name[0],
            style_feature_maps_indices_names[0],
            config,
        )
        for cnt in range(num_of_iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(
                    f"Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config.content_weight * content_loss.item():12.4f}, style loss={config.style_weight * style_loss.item():12.4f}, tv loss={config.tv_weight * tv_loss.item():12.4f}"
                )
                utils.save_and_maybe_display(
                    optimizing_img,
                    dump_path,
                    config,
                    cnt,
                    num_of_iterations,
                    should_display=False,
                )
    elif config.optimizer == "lbfgs":
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS(
            (optimizing_img,),
            max_iter=num_of_iterations,
            line_search_fn="strong_wolfe",
        )
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(
                neural_net,
                optimizing_img,
                target_representations,
                content_feature_maps_index_name[0],
                style_feature_maps_indices_names[0],
                config,
            )
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(
                    f"L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config.content_weight * content_loss.item():12.4f}, style loss={config.style_weight * style_loss.item():12.4f}, tv loss={config.tv_weight * tv_loss.item():12.4f}"
                )
                utils.save_and_maybe_display(
                    optimizing_img,
                    dump_path,
                    config,
                    cnt,
                    num_of_iterations,
                    should_display=False,
                )

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path
