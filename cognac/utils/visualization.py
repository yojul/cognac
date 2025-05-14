import io

import imageio
import matplotlib.pyplot as plt


class GifRenderingWrapper:
    def __init__(self, env, duration=0.3):
        self.env = env
        self.frames = []  # Store images in memory
        self.duration = duration  # Time per frame in GIF

    def reset(self):
        """Reset environment and clear frames."""
        self.frames = []
        return self.env.reset()

    def step(self, action_dict):
        """Perform a step and save the frame."""
        obs, reward, done, trunc, info = self.env.step(action_dict)
        self.capture_frame()
        return obs, reward, done, trunc, info

    def capture_frame(self):
        """Capture the current environment state as an image."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self.env.render(fig=fig, ax=ax)  # Your env should plot directly to `plt`

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")  # Save to buffer
        buf.seek(0)
        image = imageio.imread(buf)
        self.frames.append(image)

        buf.close()
        plt.close(fig)  # Avoid memory leaks

    def generate_gif(self, output_filename="trajectory.gif"):
        """Save the captured frames as a GIF."""
        if not self.frames:
            print("No frames captured, GIF not created.")
            return
        imageio.mimsave(output_filename, self.frames, duration=self.duration)
        print(f"GIF saved as {output_filename}")

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)  # Forward unknown methods to env
