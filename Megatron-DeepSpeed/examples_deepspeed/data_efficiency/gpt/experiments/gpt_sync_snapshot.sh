./ds_pretrain_gpt.sh model_size_config=5 checkpoint_new_thread="false" checkpoint_new_stream="false" \
                     enable_parity="false" enable_pin_memory="true" enable_sharding="false" \
                     enable_profile="true" enable_save="false" save_location="nfs" enable_snapshot="true" \
                     prealloc="true" pure_torch_save="true" get_state_dict_shape="false" \
                     save_checkpoint_in_bubble="false" fail="false" load="false"
