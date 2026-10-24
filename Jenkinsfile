pipeline {
    agent any

    parameters {
        choice(
            name: 'TEST_MODE',
            choices: ['fast', 'full'],
            description: 'fast runs deterministic PR checks; full adds network and model regression tests'
        )
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '10'))
        disableConcurrentBuilds()
        skipDefaultCheckout(true)
        timestamps()
        timeout(time: 3, unit: 'HOURS')
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PIP_NO_INPUT = '1'
        PYTHONUNBUFFERED = '1'
        VIRTUAL_ENV = "${WORKSPACE}/.venv"
    }

    stages {
        stage('Checkout') {
            steps {
                deleteDir()
                checkout scm
                sh 'git rev-parse HEAD'
            }
        }

        stage('Environment') {
            steps {
                sh '''
                    set -eux
                    python3 --version
                    python3 -m venv "${VIRTUAL_ENV}"
                    "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
                '''
            }
        }

        stage('Install') {
            steps {
                sh '''
                    set -eux
                    "${VIRTUAL_ENV}/bin/python" -m pip install -e '.[test]' 'ruff>=0.9,<1'
                    "${VIRTUAL_ENV}/bin/python" -m pip check
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '"${VIRTUAL_ENV}/bin/python" -m ruff check --select E9,F63,F7,F82 gfmbench_api tests usage_examples'
            }
        }

        stage('Fast tests') {
            steps {
                sh '''
                    mkdir -p reports
                    "${VIRTUAL_ENV}/bin/python" -m pytest \
                        tests/unit/test_caching_utils.py \
                        tests/e2e/test_smoke.py \
                        --junitxml=reports/fast-junit.xml \
                        --cov=gfmbench_api \
                        --cov-report=term-missing \
                        --cov-report=xml:reports/coverage.xml
                '''
            }
        }

        stage('Full tests') {
            when {
                expression { params.TEST_MODE == 'full' }
            }
            options {
                timeout(time: 150, unit: 'MINUTES')
            }
            steps {
                sh '''
                    mkdir -p "${WORKSPACE}/.cache/gfmbench"
                    "${VIRTUAL_ENV}/bin/python" -m pytest \
                        tests/e2e/test_download.py \
                        tests/e2e/test_heavy.py::test_heavy_sanity_regression \
                        --heavy-data-root "${WORKSPACE}/.cache/gfmbench" \
                        --junitxml=reports/full-junit.xml
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/*-junit.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**', fingerprint: true
            cleanWs(deleteDirs: true)
        }
    }
}
