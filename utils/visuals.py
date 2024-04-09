import matplotlib.pyplot as plt


def _create_image_plot(image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="grey")
    ax.set_title(title)
    return fig, ax


def show_image(image, title=""):
    fig, _ = _create_image_plot(image, title)
    fig.show()


def save_image_as_img(image, file_name, title=""):
    fig, _ = _create_image_plot(image, title)
    fig.savefig(file_name)
    plt.close()
