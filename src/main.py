import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager
from services.pdfplumberloader import PDFPlumberLoader
import traceback
import json

# Imports required by the service's model
from models.pdffile import PDFFile

settings = get_settings()

class MyService(Service):
    """
    Add the text into the PDF in the right position
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="PDF Fragmentation",
            slug="pdf-fragmentation",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="pdf",
                    type=[
                        FieldDescriptionType.APPLICATION_PDF,
                    ],
                )
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DOCUMENT_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.DOCUMENT_PROCESSING,
                ),
            ],
            has_ai=False,
        )
        self._logger = get_logger(settings)

    def process(self, data):
        try:
            # Extract the PDF file bytes from the incoming data
            raw_pdf = data["pdf"].data  # This gets the raw bytes of the PDF file
            self._logger.info("Successfully extracted PDF bytes from request")

            # Load and process the PDF using PDFPlumberLoader
            pdfLoader = PDFPlumberLoader()
            self._logger.info("Loading PDF with PDFPlumberLoader")
            try:
                pdf = PDFFile.ofBytes(raw_pdf, pdfLoader)
            except Exception as e:
                self._logger.error("Error loading PDF:\n" + traceback.format_exc())
                raise

            # Apply here the fragmentation prediction and split the PDFs
            json_prediction = json.dumps([
                1, 3, 5
            ]) # an array is enough to predict where to cut, the numbers are the pages after where we should cut.

            self._logger.info("Successfully processed PDF Text Embedding")

            # Return the result in the expected format
            return {
                "result": TaskData(data=json_prediction, type=FieldDescriptionType.APPLICATION_JSON)
            }

        except KeyError as e:
            # Handle missing "PDF" field in the request
            self._logger.error(f"Missing 'PDF' field in request: {str(e)}")
            raise ValueError("The request must include a 'PDF' field with a valid PDF file.")
        except ValueError as e:
            # Handle validation errors (e.g., no images, invalid PDF)
            self._logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            # Log any other errors and re-raise them
            self._logger.error(f"Error processing PDF: {str(e)}")
            raise

service_service: ServiceService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)

api_description = """The PDF Fragmentation service spilts the PDF into individual PDFs documents.
"""
api_summary = """Spilts the PDF into individual PDFs documents.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="PDF Fragmentation API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)