from pydantic import BaseModel, Field

class PatientSearchInput(BaseModel):
    query: str = Field(description="Medical question or search query")
    file_path: str = Field(description="Exact file path of the patient document to search")