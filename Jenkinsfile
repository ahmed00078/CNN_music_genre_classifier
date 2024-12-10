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

        stage('Run Tests') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        # Remove existing container if it exists
                        docker ps -a --filter "name=flask-api" --format "{{.ID}}" | xargs -r docker rm -f
        
                        # Start flask-api and run tests
                        docker-compose up -d flask-api
                        pytest tests/ --maxfail=1 --disable-warnings
                        docker-compose down
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