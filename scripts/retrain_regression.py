import time
from dask.distributed import wait

def retrain_regression(region):
    logger.info(f"[{region}] ▶ Retraining regression model...")

    # Start Dask cluster and patch TPOT
    client = start_dask_cluster(n_workers=2, threads_per_worker=2)
    from tpot.tpot_estimator import TPOTEstimator
    TPOTEstimator._client = client
    logger.info("✅  Dask client started and patched into TPOT")

    # Wait for workers to be fully ready
    client.wait_for_workers(n_workers=2)
    time.sleep(2)  # Extra buffer just in case

    dataset_path = f"data/processed_filled/combined_{region}_filled.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"❌  Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    X, y = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_and_resample(X, y)

    model = TPOTRegressor(
        generations=5,
        population_size=20,
        max_time_mins=30,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"[{region}] R² score: {round(score, 4)}")

    model_path = save_model(model.fitted_pipeline_, "models", region)
    logger.info(f"[{region}] ✅  Model saved: {model_path}")
    client.close()
