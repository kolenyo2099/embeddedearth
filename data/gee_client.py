"""
Google Earth Engine Client Module

Handles authentication and initialization for Earth Engine.
"""

import ee
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import gee_config


class GEEClient:
    """
    Google Earth Engine client wrapper.

    Manages authentication and provides a singleton connection to GEE.
    Supports both interactive and service account authentication.
    """

    _initialized: bool = False
    _project_id: Optional[str] = None

    @classmethod
    def initialize(cls, project_id: Optional[str] = None) -> bool:
        """
        Initialize Earth Engine.

        Args:
            project_id: GEE project ID. If None, uses config or prompts.

        Returns:
            True if initialization successful.
        """
        if cls._initialized:
            return True

        project = project_id or gee_config.project_id

        try:
            # Try to initialize with existing credentials
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()

            cls._initialized = True
            cls._project_id = project
            return True

        except ee.EEException:
            # Need to authenticate first
            try:
                ee.Authenticate()
                if project:
                    ee.Initialize(project=project)
                else:
                    ee.Initialize()

                cls._initialized = True
                cls._project_id = project
                return True

            except Exception as e:
                raise RuntimeError(f"Failed to authenticate with GEE: {e}")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if GEE is initialized."""
        return cls._initialized

    @classmethod
    def get_project_id(cls) -> Optional[str]:
        """Get the current project ID."""
        return cls._project_id
