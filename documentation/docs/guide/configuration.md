# Configuration

## 1. Add API Key

The token can be acquired under the offical mapillary developer [website](https://www.mapillary.com/developer/api-documentation).
After registration, add to key to a .env file:

```bash
touch .env
```

```
MAPILLARY_TOKEN=<TOKEN>
```

## 2. Download the docker PostgreSQL table

```bash
docker pull moritzdenk/postgis-global-streetscapes:latest
```

```bash
docker run -d \
  --name postgis-global-streetscapes \
  -p 25433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  moritzdenk/postgis-global-streetscapes:latest
```