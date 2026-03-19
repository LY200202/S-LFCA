for i in $(seq 1 3); do
    echo "========== The $i Trainning =========="
    python train_sues200.py \
        --epochs=120 \
        --height=150 \
        --classes_num=20 \
        --sample_num=3 \
        --lr=0.0001 \
        --scheduler='cosine' \
        --warmup_epochs=5
    sleep 60
done

















