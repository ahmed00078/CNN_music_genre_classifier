pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub') // DockerHub credentials in Jenkins
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[url: 'https://github.com/ahmed00078/CNN_music_genre_classifier']]
                ])
            }
        }

        stage('Test if it works') {
            steps {
                script {
                    sh 'echo "Hello, World!"'
                }
            }
        }

        stage('Build Flask API') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        docker-compose build flask-api
                    else
                        echo "docker-compose.yml not found" && exit 1
                    fi
                    '''
                }
            }
        }

        stage('Build Streamlit App') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        docker-compose build streamlit-app
                    else
                        echo "docker-compose.yml not found" && exit 1
                    fi
                    '''
                }
            }
        }

        stage('Push Docker Images to DockerHub') {
            steps {
                script {
                    sh '''
                    docker login -u $DOCKERHUB_CREDENTIALS_USR -p $DOCKERHUB_CREDENTIALS_PSW
                    docker-compose push
                    '''
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished execution.'
            script {
                sh '''
                if [ -f docker-compose.yml ]; then
                    docker-compose down || true
                fi
                '''
            }
        }
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}