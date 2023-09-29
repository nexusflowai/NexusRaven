python raven/eval/evaluator.py \
    --task_name $1 \
    --llm_name ToolAlpaca-13B \
    --agent_name TOOLALPACA \
    --hf_path Nexusflow/NexusRaven_API_evaluation \
    --standardized_queries_subset standardized_queries \
    --standardized_api_list_subset standardized_api_list
