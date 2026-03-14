# Get Started with Transfer2.5 and Predict2.5 on Brev

These setup instructions are derived from the [Get Started with Transfer2.5 and Predict2.5 on Brev](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/0a7fc52ae56709108d93e168d7bc050b79bf0804/docs/getting_started/brev/transfer2_5/transfer_and_predict_on_brev.md?plain=1). They are abbreviated and customaized to use a custom version of Cosmos2.5, that uses guided generation.

> **Authors:** Maximilian Ofir
> **Organization:** NVIDIA

## Explore Brev

NVIDIA Brev is an excellent platform for experimenting with Cosmos. Follow these steps to get started:

1. Create an account at [https://brev.nvidia.com](https://brev.nvidia.com).
2. Install the CLI as shown in [https://docs.nvidia.com/brev/cli/cli-overview](https://docs.nvidia.com/brev/cli/cli-overview).
3. Refer to the [Brev Quickstart](https://docs.nvidia.com/brev/getting-started/quickstart) to get a feel for the platform. The Brev documentation is also linked from the Brev page.

While lower spec GPUs can work for some workflows, GPUs with 80GB of VRAM are recommended for Cosmos. Also note that the Transfer 2.5 AV Multiview model requires instances with 8 or more GPUs.

## The cheat code: Launchables

[Launchables](https://docs.nvidia.com/brev/concepts/launchables) are an easy way to bundle a hardware and software environment into an easily shareable link. Once you've dialed in your Cosmos setup, a Launchable is the most convenient way to save time and share your configuration with others.

In this section, we'll walk through building a Launchable for Transfer2.5. Setting up Predict2.5 is nearly identical to the below steps. Refer to the [Predict2.5 setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md) and adjust the setup script accordingly. You can also set up both models at once.

> **Note**: Cosmos and Brev are evolving. You may encounter minor UI and other differences in the steps below as Brev changes over time.

1. Find the **Launchable** section of the Brev website.

2. Click the **Create Launchable** button.

3. Enter the Cosmos Transfer URL: [https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

4. Add a setup script. Brev will use this to install system packages.

5. If you don't need Jupyter, remove it. You can open other ports on Brev if you plan to set up a custom server.

6. Choose the desired level of compute. The screenshot below shows filtering on 8+ GPUs to run the Transfer 2.5 AV Multiview model.

7. Name your Launchable and configure access.

   You're ready to deploy! Notice the **View All Options** link, which allows you to change the compute.

8. After deploying, visit the instance page to find helpful examples of how to connect to the instance. Note the **Delete** button, which allows you to delete your instance when you're done. This can also be done with the `brev delete` CLI command. Instances that support pause and resume can be stopped from this page.

9. Connect to the instance. Once connected, follow the setup instructions provided in the [README.md](README.md) of this repository to complete your environment setup.

### Sample setup script

The sample setup script below builds a Transfer2.5 Docker image and creates another script in the home folder of your Brev environment to launch the container. Once inside the container, run the `hf auth login` command to enable checkpoint downloads. Refer to the [Transfer2.5 Downloading Checkpoints](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md#downloading-checkpoints) section for more info.

```bash
#!/bin/bash

# Install essential tools
sudo apt-get update && sudo apt-get install -y git-lfs bc curl

# Install GitHub CLI (Ubuntu 24.04+; on older distros add GitHub's APT repo first)
sudo apt-get update && sudo apt-get install -y gh

# Install Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Configure git-lfs
git lfs install
```

## Notes and Tips

- We recommend using GPUs with 80GB+ of VRAM.
- We recommend using instances with a 2 or more terabytes of storage. With less than 2 terabytes, you might run out of space.
- Don't forget to shutdown (i.e. delete) your instances when you're done.
- As of November 2025, most instances suitable for Transfer 2.5 and Predict 2.5 do not support the pause and resume (start/stop) feature.
- Note the Brev deployment time estimate when evaluating instance types (e.g. "Ready in 7minutes").
- Deployment can fail on occasion, and the driver version might not be what you expect when trying a new provider. For these reasons, set aside 3x your estimated ready time and you will be happy 😀
- Your favorite cloud provider might not always be available.
- You can change the compute for a Launchable. Here are some reasons you might want to do this:
  - ☁️ The preferred cloud provider is not available.
  - 💰 You want to save money with a different configuration.
  - 🏎️ You want to try higher specs.
