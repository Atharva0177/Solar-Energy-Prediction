#!/bin/bash
# PostgreSQL initialization script
# Creates the solar database if it doesn't exist

set -e

# Wait for PostgreSQL to be ready
until pg_isready -U "$POSTGRES_USER"; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

# Check if database exists, if not create it
if ! psql -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw "$POSTGRES_DB"; then
  echo "Creating database $POSTGRES_DB..."
  psql -U "$POSTGRES_USER" -c "CREATE DATABASE \"$POSTGRES_DB\";"
  echo "Database $POSTGRES_DB created successfully"
else
  echo "Database $POSTGRES_DB already exists"
fi
