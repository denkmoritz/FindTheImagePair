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
  -e POSTGRES_DB=gis \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 25432:5432 \
  moritzdenk/postgis-global-streetscapes:latest
```