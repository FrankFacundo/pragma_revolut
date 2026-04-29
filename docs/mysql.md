# Local MySQL Export

The data generator can write the same synthetic data to Parquet and MySQL.

```bash
docker run --name pragma-mysql -e MYSQL_ROOT_PASSWORD=pragma \
  -e MYSQL_DATABASE=pragma -p 3306:3306 -d mysql:8

pragma-generate-data \
  --out-dir data/synth \
  --users 10000 \
  --mysql-uri "mysql+pymysql://root:pragma@127.0.0.1:3306/pragma"
```

Tables:

- `pragma_users`
- `pragma_events`
- `pragma_profiles`

The Parquet files are still the training source of truth because they are
faster for local model training.
