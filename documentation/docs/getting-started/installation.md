# Installation

There are two ways to run *Image Finder*. You can either pull the project via Docker or if you want to do modifications to the code you can also run it locally.


## Via Docker


## Local Installment

To run it locally, run the following steps:

```bash
git clone https://github.com/denkmoritz/image_finder.git
cd image_finder
```

Add your [Mapillary API](https://www.mapillary.com/developer/api-documentation) key:

```bash
# in the image_finder dir
touch .env # MAPILLARY_TOKEN=<YOUR_API_KEY>
```

### Set up the Backend

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