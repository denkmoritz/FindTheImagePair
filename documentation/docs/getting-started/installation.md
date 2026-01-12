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

Add your [Mapillary API](https://www.mapillary.com/developer/api-documentation) key (not necessary):

```bash
# in the image_finder dir
touch .env # MAPILLARY_TOKEN=<YOUR_API_KEY>
```

### Set up the Backend

```bash
cd backend
```
#### Setup Images

1. Download the `images.zip` file
2. Place it in the `backend/` directory
3. Unzip the file

#### Docker Compose

```bash
docker compose up -d
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