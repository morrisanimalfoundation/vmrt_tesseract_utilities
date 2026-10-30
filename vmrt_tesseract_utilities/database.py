import datetime
import os
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import (DateTime, Float, ForeignKey, Integer, String,
                        create_engine, func, UniqueConstraint)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

load_dotenv()

"""
Database models and related utilities.
"""


class Base(DeclarativeBase):
    """
    The base class to extend for all database models.
    """
    pass


class TranscriptionInput(Base):
    """
    A table describing assets to handle in later processes, "input".
    """
    __tablename__ = 'transcription_input'
    id: Mapped[int] = mapped_column(primary_key=True)
    document_type: Mapped[str] = mapped_column(String(30))
    created: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, server_default=func.now())
    input_file: Mapped[Optional[str]] = mapped_column(String(255))
    assets: Mapped[List['TranscriptionOutput']] = relationship(
        back_populates='transcription_input', cascade='all, delete-orphan'
    )
    derived_metadata: Mapped[List['TranscriptionMetadata']] = relationship(
        back_populates='transcription_input', cascade='all, delete-orphan'
    )
    __table_args__ = (
        UniqueConstraint(
            "input_file",
            "document_type",
            name="uq_transcription_input_file_document_type",
        ),
    )

    def __repr__(self) -> str:
        return f'Transcription Input (id={self.id!r}, document_type={self.document_type!r})'


class TranscriptionOutput(Base):
    """
    A table describing assets produced by our various processes, "output".
    """
    __tablename__ = 'transcription_output'
    id: Mapped[int] = mapped_column(primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey('transcription_input.id'))
    transcription_input: Mapped['TranscriptionInput'] = relationship(back_populates='assets')
    ocr_output_file: Mapped[Optional[str]] = mapped_column(String(255))
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float)
    list_replacement_output_file: Mapped[Optional[str]] = mapped_column(String(255))
    pii_scrubber_output_file: Mapped[Optional[str]] = mapped_column(String(255))
    pii_scrubber_confidence_file: Mapped[Optional[str]] = mapped_column(String(255))

    def __repr__(self) -> str:
        return f'Transcription Output (id={self.id})!r'


class TranscriptionMetadata(Base):
    """
    A table containing metdata we've mined from the various processes.
    """
    __tablename__ = 'transcription_metadata'
    id: Mapped[int] = mapped_column(primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey('transcription_input.id'))
    transcription_input: Mapped['TranscriptionInput'] = relationship(back_populates='derived_metadata')
    subject_id: Mapped[Optional[str]] = mapped_column(String(11))
    year_in_study: Mapped[Optional[int]] = mapped_column(Integer)
    visit_date: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    extracted_date: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)

    def __repr__(self) -> str:
        return f'Transcription Metadata (id={self.id})!r'


class StringReplacementPass(Base):
    """
    Records the configuration and identifying information for a string-replacement pass.
    """
    __tablename__ = "string_replacement_pass"

    id: Mapped[int] = mapped_column(primary_key=True)
    pass_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    replacement_string: Mapped[str] = mapped_column(String(255), nullable=False)
    source_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    key_column: Mapped[str] = mapped_column(String(255), nullable=False)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=func.now()
    )


class StringReplacementCompletion(Base):
    """
    Records that a transcription output completed a particular string-replacement pass.
    """
    __tablename__ = "string_replacement_completion"

    id: Mapped[int] = mapped_column(primary_key=True)
    output_id: Mapped[int] = mapped_column(
        ForeignKey("transcription_output.id"), nullable=False
    )
    pass_id: Mapped[int] = mapped_column(
        ForeignKey("string_replacement_pass.id"), nullable=False
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "output_id",
            "pass_id",
            name="uq_string_replacement_completion",
        ),
    )


def get_database_session(**kwargs) -> sessionmaker:
    """
    Gets the database connection.

    Returns
    -------
    _engine : sqlalchemy.engine.Engine
      The database connection.
    """
    sql_url = os.getenv('SQL_CONNECTION_STRING')
    if sql_url is None:
        RuntimeError('SQL_CONNECTION_STRING environment variable is not set.')
    engine = create_engine(sql_url, **kwargs)
    return sessionmaker(engine)
