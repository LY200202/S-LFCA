for i in $(seq 1 2); do
    echo "========== The $i Trainning =========="
    python train_university.py \
        --epochs=120 \
        --classes_num=20 \
        --sample_num=3 \
        --lr=0.0001 \
        --scheduler='cosine' \
        --warmup_epochs=5
    sleep 120
done
















