# EasyMDM Advanced - Flask Web Interface

A modern and intuitive web interface for EasyMDM Advanced, providing full master data management (MDM) capabilities with a complete user experience.

## 🌟 Features

### Modern User Interface

* **Responsive Design**: Works on all screens (desktop, tablet, mobile)
* **Drag & Drop**: Upload CSV files easily by dragging and dropping
* **Smooth Animations**: CSS transitions and animations for a pleasant experience
* **Modern Theme**: Gradients and attractive visual elements

### Complete MDM Features

* **Interactive Configuration**: Configure MDM parameters via graphical interface
* **5 Predefined Examples**: Full demonstrations of EasyMDM capabilities
* **Real-Time Monitoring**: Task progress with visual progress bar
* **Detailed Logs**: Live execution logs with terminal-style display

### Data Management

* **Automatic CSV Analysis**: Auto-detect columns and generate statistics
* **File Validation**: Check format and integrity
* **Sample Generation**: Auto-create test data
* **Results Export**: Download processed files

---

## 🚀 Installation and Launch

### Method 1: Automatic Launch (Recommended)

```bash
# Launch the app directly
python run_app.py
```

The script will automatically:

* Check and install dependencies
* Create necessary folders
* Launch the application
* Open the browser automatically

### Method 2: Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements_flask.txt

# 2. Launch the app
python app.py
```

### Method 3: Virtual Environment

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements_flask.txt

# 4. Launch the app
python app.py
```

---

## 📱 Usage

### 1. Start

Application available at: `http://localhost:5000`

### 2. Upload Data

* **Drag & Drop**: Drag your CSV file into the designated area
* **Manual Select**: Click the area to open the file selector
* **Auto Sample**: Click "Create Sample" for test data

### 3. MDM Configuration

Once the file is loaded, configure:

* **Blocking Columns**: Fields used to group records
* **Blocking Method**: Exact or Fuzzy
* **Thresholds**: Review and auto-merge
* **Output Folder**: Directory for results

### 4. Predefined Examples

#### Example 1: Basic CSV

* Process CSV with minimal configuration
* Demonstrates basic features
* Simulated French customer data

#### Example 2: Advanced Similarity

* Test and compare similarity algorithms
* Methods: Jaro-Winkler, Levenshtein, Exact, Cosine
* Column-specific recommendations

#### Example 3: PostgreSQL Configuration

* Advanced database setup
* Connection parameters and optimization
* Complex survivorship rules

#### Example 4: Performance Optimization

* Optimized processing for large volumes
* Parallelization and memory settings
* Detailed performance metrics

#### Example 5: Custom Survivorship

* Advanced survivorship strategies
* Multiple priority conditions
* Intelligent conflict resolution

### 5. Tracking & Results

* **Progress Bar**: Visual processing status
* **Real-Time Logs**: Detailed process messages
* **Statistics**: Full processing metrics
* **Download**: Export generated files

---

## 🏗️ Architecture

### File Structure

```
easymdm-flask/
│
├── app.py                 # Main Flask app
├── run_app.py             # Auto-launch script
├── requirements_flask.txt # Python dependencies
├── README.md              # Documentation
│
├── templates/
│   └── index.html         # Main user interface
│
├── static/
│   └── css/
│       └── custom.css     # Custom styles
│
├── uploads/               # Uploaded files (auto-created)
├── outputs/               # Generated results (auto-created)
└── configs/               # Saved configurations (auto-created)
```

### Technologies Used

* **Backend**: Flask 2.3+ (Python)
* **Frontend**: Bootstrap 5.3, HTML5, CSS3, JavaScript ES6
* **UI/UX**: CSS animations, responsive design, Bootstrap icons
* **Data Handling**: Pandas, NumPy for CSV processing
* **Processing**: Threading for asynchronous tasks

---

## 🔧 Configuration

### Environment Variables

```python
# Flask configuration (in app.py)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max 50MB
```

### Customizable Parameters

* **Port**: Default 5000, changeable in `run_app.py`
* **Host**: Default 127.0.0.1, changeable for network access
* **Max File Size**: 50MB per file, configurable
* **Directories**: Custom paths for uploads/outputs

---

## 🎨 Customization

### User Interface

All custom styles in `static/css/custom.css`:

* Colors & gradients
* Animations & transitions
* Responsive design
* Custom UI components

### Adding Features

1. **Flask Route**: Add in `app.py`

```python
@app.route('/new_feature')
def new_feature():
    # Processing logic
    return jsonify({'success': True})
```

2. **Interface**: Update `templates/index.html`
3. **Styles**: Add CSS in `static/css/custom.css`

---

## 📊 Monitoring & Logs

### Task Tracking

* **Real-Time Status**: Progress, status, logs
* **Temporary Storage**: Global variables for tracking
* **Auto Cleanup**: Memory management for completed tasks

### Execution Logs

* **Terminal Style**: Console-like display with colors
* **Timestamps**: Each message is time-stamped
* **Auto Scroll**: Shows latest messages
* **History**: Keeps last 50 logs per task

---

## 🔒 Security

### File Uploads

* **Validated Extensions**: Only `.csv` allowed
* **Secure Names**: Using `secure_filename()`
* **Max Size**: 50MB per file
* **Timestamped Names**: Prevent conflicts

### Data Validation

* **CSV Format**: Checks structure & encoding
* **Parameters**: Server-side validation
* **Paths**: Directory traversal protection

---

## 🚀 Deployment

### Development

```bash
python run_app.py
# or
python app.py
```

### Production

Use a WSGI server like Gunicorn:

```bash
# Install
pip install gunicorn

# Run
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_flask.txt .
RUN pip install -r requirements_flask.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 📝 License

MIT License. See LICENSE file for details.

## 🤝 Contribution

Contributions welcome! To contribute:

1. Fork the project
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Create a Pull Request

## 📞 Support

For questions or issues:

* Open an issue on GitHub
* Check EasyMDM documentation
* Review error logs in the console

---

**EasyMDM Advanced Flask** – Modern web interface for master data management 🚀

---

