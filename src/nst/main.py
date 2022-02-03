import argparse
import os


from nst import neural_style_transfer


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), "data")
    content_images_dir = os.path.join(default_resource_dir, "content-images")
    style_images_dir = os.path.join(default_resource_dir, "style-images")
    output_img_dir = os.path.join(default_resource_dir, "output-images")
    img_format = (4, ".jpg")  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content_img_name", type=str, help="content image name", default="figures.jpg"
    )
    parser.add_argument(
        "--style_img_name",
        type=str,
        help="style image name",
        default="vg_starry_night.jpg",
    )
    parser.add_argument(
        "--height", type=int, help="height of content and style images", default=400
    )

    parser.add_argument(
        "--content_weight",
        type=float,
        help="weight factor for content loss",
        default=1e5,
    )
    parser.add_argument(
        "--style_weight", type=float, help="weight factor for style loss", default=3e4
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        help="weight factor for total variation loss",
        default=1e0,
    )

    parser.add_argument(
        "--optimizer", type=str, choices=["lbfgs", "adam"], default="lbfgs"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--model", type=str, choices=["vgg16", "vgg19"], default="vgg19"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["random", "content", "style"],
        default="content",
    )
    parser.add_argument(
        "--saving_freq",
        type=int,
        help="saving frequency for intermediate images (-1 means only final)",
        default=-1,
    )
    args = parser.parse_args()

    # some values of weights that worked for figures.jpg, vg_starry_night.jpg (starting point for finding good images)
    # once you understand what each one does it gets really easy -> also see README.md

    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # just wrapping settings into a dictionary
    args_as_dict = {var: getattr(args, var) for var in vars(args)}
    optimization_config = neural_style_transfer.Config(**args_as_dict)
    optimization_config.content_images_dir = content_images_dir
    optimization_config.style_images_dir = style_images_dir
    optimization_config.output_img_dir = output_img_dir
    optimization_config.img_format = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer.transfer_style(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
