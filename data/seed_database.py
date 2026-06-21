"""
Seed script to create and populate the sample company SQLite database.

Uses Faker to generate realistic fake data at scale.
Run this once to generate data/company.db.
"""

import sqlite3
import os
import random

from faker import Faker

DB_PATH = os.path.join(os.path.dirname(__file__), "company.db")

fake = Faker()
Faker.seed(42)
random.seed(42)


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
            position TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL,
            email TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            start_date TEXT NOT NULL,
            deadline TEXT NOT NULL,
            budget REAL NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
    """)

    # Seed 10 departments
    departments = [
        (1, "Engineering", 1200000),
        (2, "Marketing", 450000),
        (3, "Sales", 600000),
        (4, "Human Resources", 300000),
        (5, "Finance", 500000),
        (6, "Product", 700000),
        (7, "Customer Support", 350000),
        (8, "Legal", 400000),
        (9, "Operations", 550000),
        (10, "Research & Development", 900000),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?)", departments)

    # Seed 2000 employees
    positions = [
        "Junior Developer", "Senior Developer", "Lead Developer", "Architect",
        "Analyst", "Senior Analyst", "Manager", "Director", "VP",
        "Coordinator", "Specialist", "Consultant", "Intern", "Team Lead",
    ]

    employees = []
    for i in range(1, 2001):
        dept_id = random.randint(1, 10)
        position = random.choice(positions)
        salary = random.randint(35000, 150000)
        hire_date = fake.date_between(start_date="-5y", end_date="today").isoformat()
        email = fake.email()
        name = fake.name()
        employees.append((i, name, dept_id, position, salary, hire_date, email))

    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)

    # Seed 50 projects
    statuses = ["planned", "in_progress", "completed", "on_hold", "cancelled"]
    project_names = [
        "Cloud Migration", "API Redesign", "Mobile App v2", "Data Pipeline",
        "Security Audit", "Brand Refresh", "CRM Integration", "AI Chatbot",
        "Performance Optimization", "Infrastructure Upgrade", "User Research",
        "Payment Gateway", "Analytics Dashboard", "DevOps Pipeline",
        "Customer Portal", "Inventory System", "Email Campaign", "SEO Overhaul",
        "Compliance Review", "Training Program", "Office Expansion",
        "Vendor Management", "Quality Assurance", "Knowledge Base",
        "Recruitment Drive", "Budget Planning", "Market Expansion",
        "Product Launch", "Partnership Program", "Internal Tooling",
        "Disaster Recovery", "Green Initiative", "Accessibility Audit",
        "Localization", "Supply Chain Optimization", "Loyalty Program",
        "Social Media Strategy", "Onboarding Revamp", "Tech Debt Reduction",
        "Customer Feedback System", "Real-time Reporting", "Microservices Split",
        "Load Testing", "Documentation Overhaul", "Mentorship Program",
        "Open Source Contribution", "Hackathon Planning", "Cost Reduction",
        "Feature Flagging", "AB Testing Platform",
    ]

    projects = []
    for i in range(1, 51):
        dept_id = random.randint(1, 10)
        status = random.choice(statuses)
        start_date = fake.date_between(start_date="-2y", end_date="today").isoformat()
        deadline = fake.date_between(start_date="today", end_date="+1y").isoformat()
        budget = random.randint(10000, 500000)
        name = project_names[i - 1]
        projects.append((i, name, dept_id, status, start_date, deadline, budget))

    cursor.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)", projects)

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")
    print(f"  - {len(departments)} departments")
    print(f"  - {len(employees)} employees")
    print(f"  - {len(projects)} projects")


if __name__ == "__main__":
    create_database()
