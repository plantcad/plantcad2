for model in pcv2-l24-d0768 pcv2-l48-d1024 pcv2-l48-d1536; do
    CUDA_VISIBLE_DEVICES=1 python main.py -length 512 -outLogit test_${model}_512_logit.tsv \
        -model 'maize-genetics/'$model \
        -batchSize 256 \
        -tokenIdx 255
done


for model in pcv2-l24-d0768 pcv2-l48-d1024 pcv2-l48-d1536; do
    CUDA_VISIBLE_DEVICES=0 python main.py -length 1024 -outLogit test_${model}_1024_logit.tsv \
        -model 'maize-genetics/'$model \
        -batchSize 128 \
        -tokenIdx 511
done

for model in pcv2-l24-d0768 pcv2-l48-d1024; do
    CUDA_VISIBLE_DEVICES=0 python main.py -length 2048 -outLogit test_${model}_2048_logit.tsv \
        -model 'maize-genetics/'$model \
        -batchSize 64 \
        -tokenIdx 1023
done

