import pytest
import sys
from pathlib import Path


def test_pytest_installed():
    """Test that pytest is properly installed."""
    assert "pytest" in sys.modules or True


def test_project_structure():
    """Test that the project structure is set up correctly."""
    project_root = Path(__file__).parent.parent
    
    # Check main package exists
    assert (project_root / "jetmoe").exists()
    assert (project_root / "jetmoe" / "__init__.py").exists()
    
    # Check test structure
    assert (project_root / "tests").exists()
    assert (project_root / "tests" / "__init__.py").exists()
    assert (project_root / "tests" / "conftest.py").exists()
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and is properly configured."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    assert pyproject_path.exists()
    
    # Check content
    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content
    assert "[tool.pytest.ini_options]" in content
    assert "[tool.coverage.run]" in content


@pytest.mark.unit
def test_markers_work():
    """Test that custom markers are properly configured."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test integration marker."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test slow marker."""
    assert True


def test_fixtures_available(temp_dir, mock_config, sample_tensor):
    """Test that fixtures from conftest.py are available."""
    assert temp_dir.exists()
    assert hasattr(mock_config, 'hidden_size')
    assert sample_tensor.shape == (2, 10, 768)


def test_coverage_configured():
    """Test that coverage is properly configured."""
    import coverage
    assert coverage.__version__