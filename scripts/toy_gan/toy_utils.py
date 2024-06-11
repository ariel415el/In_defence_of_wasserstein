from matplotlib import pyplot as plt


def draw_points(points, save_path, ref_data=None, **kwargs):
    # Plotting
    # plt.figure(figsize=(15, 15))
    if ref_data is not None:
        plt.scatter(ref_data[:, 0], ref_data[:, 1], color="b", label="Real data", alpha=0.5, s=3, **kwargs)
    plt.scatter(points[:, 0], points[:, 1], color="r", label="Fake data", alpha=0.5, s=3, **kwargs)
    plt.axis('equal')  # Set equal aspect ratio
    plt.title('Points on a Circle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.clf()