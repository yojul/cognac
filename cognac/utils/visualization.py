import io

import imageio
import matplotlib.pyplot as plt
from pettingzoo import ParallelEnv


class GifRenderingWrapper:
    """A wrapper for PettingZoo environments that captures and saves the environment's
    visual trajectory as an animated GIF.

    .. warning::
        The gif rendering wrapper is still a work in progress and rely on proper r
        endering
        capability of environment. We are working on it to provide working solution to
        generate gif from trajectories.

    This is useful for visualizing multi-agent interactions over time, particularly in
    notebook-based workflows or for producing documentation and demos.

    Parameters
    ----------
    env : pettingzoo.ParallelEnv
        The environment to wrap. It must implement a `render(fig, ax)` method that
        draws the environment state using Matplotlib.
    duration : float, optional
        Duration (in seconds) per frame in the output GIF. Default is 0.3.

    Attributes
    ----------
    env : pettingzoo.ParallelEnv
        The original environment instance being wrapped.
    frames : list
        A list of image frames (as numpy arrays) captured during the environment
        rollout.
    duration : float
        Time interval between frames in the generated GIF.
    """

    def __init__(self, env: ParallelEnv, duration: float = 0.3):
        self.env = env
        self.frames = []  # Store images in memory
        self.duration = duration  # Time per frame in GIF

    def reset(self):
        """Reset the environment and clear any previously stored frames.

        Returns
        -------
        Any
            The output of the wrapped environment's `reset` method.
        """
        self.frames = []
        return self.env.reset()

    def step(self, action_dict):
        """Step the environment forward using the provided actions and capture the
        frame.

        Parameters
        ----------
        action_dict : dict
            A dictionary mapping agent names to their actions.

        Returns
        -------
        tuple
            A tuple (obs, reward, done, trunc, info) as returned by
            the wrapped environment.
        """
        obs, reward, done, trunc, info = self.env.step(action_dict)
        self.capture_frame()
        return obs, reward, done, trunc, info

    def capture_frame(self):
        """Capture the current environment state as an image."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self.env.render(fig=fig, ax=ax)  # Env should plot directly to `plt`

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")  # Save to buffer
        buf.seek(0)
        image = imageio.imread(buf)
        self.frames.append(image)

        buf.close()
        plt.close(fig)  # Avoid memory leaks

    def generate_gif(self, output_filename: str = "trajectory.gif"):
        """Generate and save a GIF from the captured frames.

        Parameters
        ----------
        output_filename : str, optional
            Name of the output GIF file (default is "trajectory.gif").

        Returns
        -------
        None
        """
        if not self.frames:
            print("No frames captured, GIF not created.")
            return
        imageio.mimsave(output_filename, self.frames, duration=self.duration)
        print(f"GIF saved as {output_filename}")

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment.

        This allows the wrapper to behave transparently as the original environment
        for attributes and methods not explicitly overridden.

        Parameters
        ----------
        name : str
            Name of the attribute or method being accessed.

        Returns
        -------
        Any
            The corresponding attribute from the wrapped environment.
        """
        return getattr(self.env, name)  # Forward unknown methods to env
