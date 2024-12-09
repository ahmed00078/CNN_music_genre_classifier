### **1. Install and Run Jenkins**
#### **Option A: Using Docker (Recommended for Simplicity)**
1. Install Docker if you haven’t already: [Get Docker](https://docs.docker.com/get-docker/).
2. Run Jenkins:
   ```bash
   docker run -d --name jenkins -p 8080:8080 -p 50000:50000 \
    -v jenkins_home:/var/jenkins_home \
    -v //var/run/docker.sock:/var/run/docker.sock \
    --user root \
    jenkins/jenkins:lts
   ```
   - This will start Jenkins on `http://localhost:8080`.

3. Retrieve the admin password:
   ```bash
   docker logs jenkins
   ```
   Copy the password to log in to Jenkins.

#### **Option B: Install Locally (Windows Example)**
1. Download the Jenkins installer for Windows: [Jenkins Windows Installer](https://www.jenkins.io/download/).
2. Follow the installation wizard.
3. Start Jenkins and navigate to `http://localhost:8080`.

---

### **2. Configure Jenkins After Installation**
1. Log in with the admin password you retrieved.
2. Choose **Install Suggested Plugins** during setup.
3. Set up an admin user (you can reuse the default admin for simplicity).

---

### **3. Install Required Plugins**
1. Go to **Manage Jenkins** > **Plugins** > **Available Plugins**.
2. Search for and install the following:
   - **Pipeline** (already included in most setups)
   - **Git**
   - **Docker Pipeline**
   - **NodeJS** (if needed for Streamlit dependencies)

---

### **4. Create a New Jenkins Pipeline Job**
1. Click **New Item** in Jenkins.
2. Name your job (e.g., `Music Genre Classifier`).
3. Select **Pipeline** and click **OK**.
4. In the job configuration, set the **Pipeline Script from SCM**:
   - **SCM**: Git
   - **Repository URL**: Your project’s Git repository.
   - **Branch**: The branch you want to track (e.g., `main`).

---

### **5. Write the `Jenkinsfile`**
The `Jenkinsfile` defines your CI pipeline steps.

#### **Basic Example for Your Project:**
Place this `Jenkinsfile` in your repository's root:

```groovy
pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub') // Define DockerHub credentials in Jenkins
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Flask API') {
            steps {
                script {
                    sh 'docker-compose build flask-api'
                }
            }
        }

        stage('Build Streamlit App') {
            steps {
                script {
                    sh 'docker-compose build streamlit-app'
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Adjust this if your project has tests
                    sh 'docker-compose up -d flask-api'
                    sh 'pytest tests/ --maxfail=1 --disable-warnings'
                    sh 'docker-compose down'
                }
            }
        }

        stage('Push Docker Images to DockerHub') {
            steps {
                script {
                    sh 'docker login -u $DOCKERHUB_CREDENTIALS_USR -p $DOCKERHUB_CREDENTIALS_PSW'
                    sh 'docker-compose push'
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished execution.'
            sh 'docker-compose down' // Ensure cleanup
        }
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
```

---

### **6. Set Up DockerHub Credentials in Jenkins**
1. Go to **Manage Jenkins** > **Credentials** > **Global**.
2. Add a new credential with:
   - **Username**: Your DockerHub username.
   - **Password**: Your DockerHub password.
   - **ID**: `dockerhub` (as referenced in the `Jenkinsfile`).

---

### **7. Test Your Pipeline**
1. Save the job configuration.
2. Trigger the pipeline manually:
   - Click **Build Now** in Jenkins.
3. Monitor the build progress:
   - Check the **Console Output** for errors or progress.

---

### **8. Automate Builds with Triggers**
1. Go to your pipeline configuration.
2. Under **Build Triggers**, choose one:
   - **Poll SCM**: Jenkins checks for changes periodically (e.g., every 5 minutes):
     ```
     H/5 * * * *
     ```
   - **GitHub Webhook**: Set up a webhook in your GitHub repository to trigger Jenkins builds.

---

### **9. Optional: Add Notifications**
Install and configure notification plugins to inform you of build statuses:
- **Slack**: Notify your Slack channel of build results.
- **Email Extension**: Send email alerts.