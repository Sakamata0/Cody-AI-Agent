"""
Seed script to create and populate the sample company SQLite database.

Run this once to generate data/company.db with fake company data.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "company.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        DROP TABLE IF EXISTS projects;
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS departments;

        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL NOT NULL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            deadline TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
    """)

    # Seed departments
    departments = [
        (1, "Engineering", 500000),
        (2, "Marketing", 200000),
        (3, "Sales", 300000),
        (4, "Human Resources", 150000),
        (5, "Finance", 250000),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?)", departments)

    # Seed employees
    employees = [
        (1, "Alice Martin", 1, 75000, "2022-03-15"),
        (2, "Bob Johnson", 1, 82000, "2021-07-01"),
        (3, "Charlie Brown", 2, 60000, "2023-01-10"),
        (4, "Diana Ross", 3, 70000, "2022-06-20"),
        (5, "Eve Wilson", 1, 90000, "2020-11-05"),
        (6, "Frank Miller", 4, 55000, "2023-04-12"),
        (7, "Grace Lee", 2, 65000, "2022-09-30"),
        (8, "Henry Davis", 3, 72000, "2021-12-01"),
        (9, "Iris Chen", 5, 78000, "2022-02-14"),
        (10, "Jack Taylor", 1, 85000, "2021-05-22"),
        (11, "Karen White", 5, 80000, "2020-08-17"),
        (12, "Liam Harris", 3, 68000, "2023-03-01"),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)

    # Seed projects
    projects = [
        (1, "Cloud Migration", 1, "in_progress", "2025-09-30"),
        (2, "Brand Redesign", 2, "completed", "2025-03-15"),
        (3, "Q3 Sales Campaign", 3, "in_progress", "2025-08-31"),
        (4, "Employee Portal", 4, "planned", "2025-12-01"),
        (5, "AI Integration", 1, "in_progress", "2025-10-15"),
        (6, "Budget Automation", 5, "planned", "2025-11-30"),
    ]
    cursor.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", projects)

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")
    print(f"  - {len(departments)} departments")
    print(f"  - {len(employees)} employees")
    print(f"  - {len(projects)} projects")


if __name__ == "__main__":
    create_database()
