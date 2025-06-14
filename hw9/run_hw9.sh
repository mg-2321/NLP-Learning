#!/bin/bash
#!/bin/bash
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py \
    --template_method basic \
    --prompt_method basic \
    --model_name allenai/OLMo-2-0425-1B \
    --num_in_context_examples 0

/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method basic --prompt_method basic --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method basic --prompt_method complex --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method basic --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method complex --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method basic --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 4
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method complex --model_name allenai/OLMo-2-0425-1B --num_in_context_examples 4
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method basic --prompt_method basic --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method basic --prompt_method complex --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method basic --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method complex --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 0
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method basic --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 4
/mnt/dropbox/24-25/574/env-new/bin/python olmo.py --template_method complex --prompt_method complex --model_name allenai/OLMo-2-0425-1B-SFT --num_in_context_examples 4
