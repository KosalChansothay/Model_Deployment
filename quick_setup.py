#!/usr/bin/env python3
"""
Quick Setup Script for Iris API
Creates directory structure and verifies all files
"""
import os
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['static', 'templates']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"✓ Created directory: {d}")
        else:
            print(f"✓ Directory exists: {d}")

def check_files():
    """Check if all required files exist"""
    required_files = {
        'app.py': 'Main FastAPI application',
        'train_model.py': 'Model training script',
        'requirements.txt': 'Python dependencies',
        'Dockerfile': 'Docker configuration',
        'docker-compose.yml': 'Docker Compose configuration',
        'static/style.css': 'CSS stylesheet',
        'templates/index.html': 'HTML template'
    }
    
    print("\n" + "=" * 60)
    print("CHECKING FILES")
    print("=" * 60)
    
    missing = []
    for file, description in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file:<25} ({size:>8} bytes) - {description}")
        else:
            print(f"✗ {file:<25} MISSING - {description}")
            missing.append(file)
    
    return missing

def check_model():
    """Check if model.pkl exists"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL")
    print("=" * 60)
    
    if os.path.exists('model.pkl'):
        size = os.path.getsize('model.pkl')
        print(f"✓ model.pkl exists ({size} bytes)")
        return True
    else:
        print("✗ model.pkl NOT FOUND")
        print("\n  → Run: python train_model.py")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    packages = [
        'fastapi',
        'uvicorn', 
        'pydantic',
        'sklearn',
        'pandas',
        'numpy',
        'jinja2'
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    return missing

def print_directory_structure():
    """Print the directory structure"""
    print("\n" + "=" * 60)
    print("DIRECTORY STRUCTURE")
    print("=" * 60)
    print("""
iris-api/
├── app.py                 # FastAPI application
├── train_model.py         # Model training
├── model.pkl             # Trained models
├── requirements.txt      # Dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose config
├── static/
│   └── style.css        # CSS stylesheet
└── templates/
    └── index.html       # HTML template
    """)

def print_next_steps(has_model, missing_files, missing_deps):
    """Print next steps based on what's missing"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    step = 1
    
    if missing_files:
        print(f"\n{step}. Create missing files:")
        for file in missing_files:
            print(f"   - {file}")
        step += 1
    
    if missing_deps:
        print(f"\n{step}. Install missing dependencies:")
        print("   pip install -r requirements.txt")
        step += 1
    
    if not has_model:
        print(f"\n{step}. Train the models:")
        print("   python train_model.py")
        step += 1
    
    if not missing_files and not missing_deps and has_model:
        print("\n✅ Everything is ready!")
        print("\nRun the application:")
        print("   python app.py")
        print("\nOr use Docker:")
        print("   docker-compose up -d")
        print("\nThen open: http://localhost:8000")
    else:
        print(f"\n{step}. After completing above steps, run:")
        print("   python app.py")

def test_static_file_access():
    """Test if static files are accessible"""
    print("\n" + "=" * 60)
    print("TESTING FILE ACCESS")
    print("=" * 60)
    
    if os.path.exists('static/style.css'):
        try:
            with open('static/style.css', 'r') as f:
                content = f.read()
                if len(content) > 0:
                    print(f"✓ style.css is readable ({len(content)} characters)")
                    # Check if it contains CSS
                    if 'body' in content or 'font-family' in content:
                        print("✓ style.css contains valid CSS")
                    else:
                        print("⚠ style.css might be empty or invalid")
                else:
                    print("✗ style.css is empty")
        except Exception as e:
            print(f"✗ Error reading style.css: {e}")
    
    if os.path.exists('templates/index.html'):
        try:
            with open('templates/index.html', 'r') as f:
                content = f.read()
                if len(content) > 0:
                    print(f"✓ index.html is readable ({len(content)} characters)")
                    # Check for CSS link
                    if '/static/style.css' in content:
                        print("✓ index.html links to /static/style.css")
                    else:
                        print("⚠ index.html might not link to CSS")
                else:
                    print("✗ index.html is empty")
        except Exception as e:
            print(f"✗ Error reading index.html: {e}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("IRIS API - QUICK SETUP & VERIFICATION")
    print("=" * 60)
    print(f"\nCurrent directory: {os.getcwd()}\n")
    
    # Create directories
    create_directories()
    
    # Check files
    missing_files = check_files()
    
    # Check model
    has_model = check_model()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Test file access
    test_static_file_access()
    
    # Print directory structure
    print_directory_structure()
    
    # Print next steps
    print_next_steps(has_model, missing_files, missing_deps)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()