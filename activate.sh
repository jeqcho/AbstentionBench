if [[ $OSTYPE != "darwin"* ]]; then
  source /etc/profile.d/modules.sh

  module purge

  module load cuda/12.1
  module load cudnn/v9.0.0.312-cuda.12.3
  module load NCCL/2.21.5-1-cuda.12.2
fi

# Load our shared HF token
export HF_TOKEN=`cat /large_experiments/robust_vlm/abstention-bench/huggingface/hf_token`
# A shared HF directory created consistent Permission Denied issues.
# export HF_HOME="/large_experiments/robust_vlm/abstention-bench/huggingface"
export AZURE_GPT4O_API_KEY=`cat .secrets/azure_gpt4o_api_key`
export AZURE_O1_PREVIEW_API_KEY=`cat .secrets/azure_o1_preview_api_key`
export AZURE_O1_API_KEY=`cat .secrets/azure_o1_api_key`
export GOOGLE_GENAI_API_KEY=`cat .secrets/google_genai_api_key`

mamba activate abstention-bench