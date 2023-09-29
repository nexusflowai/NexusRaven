python raven/eval/evaluator.py \
    --task_name $1 \
    --llm_name openai_gpt3.5_instruct \
    --agent_name SIMPLE \
    --hf_path Nexusflow/NexusRaven_API_evaluation \
    --standardized_queries_subset standardized_queries \
    --standardized_api_list_subset standardized_api_list
