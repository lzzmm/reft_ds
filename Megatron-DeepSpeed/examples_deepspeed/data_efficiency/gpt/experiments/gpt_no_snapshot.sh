./ds_pretrain_gpt.sh model_size_config=3 checkpoint_new_thread="true" checkpoint_new_stream="true" \
                     enable_parity="false" enable_pin_memory="true" enable_sharding="true" \
                     enable_profile="true" enable_save="false" save_location="nfs" enable_snapshot="false" \
                     prealloc="true" pure_torch_save="false" get_state_dict_shape="false" \
                     save_checkpoint_in_bubble="false" fail="false" load="false" enable_cpu_optimizer="false" \
                     original_load="false" double_snapshot="false" failed_ranks="" enable_non_blocking="true"
