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
