# Installation

There are two ways to run *Find The Image Pair*. You can either pull the project via Docker or if you want to do modifications to the code you can also run it locally.


## Via Docker

**COMING SOON**

## Local Installment

To run it locally, run the following steps:

```bash
https://github.com/denkmoritz/FindTheImagePair.git
cd FindTheImagePair
```

Add your [Mapillary API](https://www.mapillary.com/developer/api-documentation) key:

```bash
# in the image_finder dir
touch .env # MAPILLARY_TOKEN=<YOUR_API_KEY>
```

### Set up the Backend (Used Python3.11)

```bash
cd backend
python -m venv venv
pip install -r requirements.txt
```

To run the Backend:

```bash
uvicorn main:app --host 0.0.0.0 --reload
```

### Set up the Frontend

```bash
cd frontend
npm install
```

To run the Frontend:

```bash
npm run dev
```

## Pull the PostgreSQL DB from Docker

```bash
docker pull moritzdenk/postgis-global-streetscapes:latest
```

```bash
docker run -d \
  --name postgis-global-streetscapes \
  -p 25433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  denkmoritz/postgis-global-streetscapes:latest
```