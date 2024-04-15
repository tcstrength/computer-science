CREATE TABLE Books (
  book_id INT PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(255) NOT NULL,
  author_id INT NOT NULL,
  genre_id INT NOT NULL,
  publication_year INT,
  isbn VARCHAR(13) UNIQUE,
  FOREIGN KEY (author_id) REFERENCES Authors(author_id),
  FOREIGN KEY (genre_id) REFERENCES Genres(genre_id)
);

CREATE TABLE Authors (
  author_id INT PRIMARY KEY AUTO_INCREMENT,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  birth_year INT
);

CREATE TABLE Genres (
  genre_id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(50) NOT NULL,
  description TEXT
);

CREATE TABLE Members (
  member_id INT PRIMARY KEY AUTO_INCREMENT,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  address VARCHAR(255),
  phone_number VARCHAR(20),
  email VARCHAR(100) UNIQUE
);

CREATE TABLE Loans (
  loan_id INT PRIMARY KEY AUTO_INCREMENT,
  book_id INT NOT NULL,
  member_id INT NOT NULL,
  loan_date DATE NOT NULL,
  due_date DATE NOT NULL,
  return_date DATE,
  FOREIGN KEY (book_id) REFERENCES Books(book_id),
  FOREIGN KEY (member_id) REFERENCES Members(member_id)
);

CREATE TABLE Fines (
  fine_id INT PRIMARY KEY AUTO_INCREMENT,
  loan_id INT NOT NULL,
  fine_amount DECIMAL(10,2) NOT NULL,
  payment_date DATE,
  FOREIGN KEY (loan_id) REFERENCES Loans(loan_id)
);

CREATE TABLE Reservations (
  reservation_id INT PRIMARY KEY AUTO_INCREMENT,
  book_id INT NOT NULL,
  member_id INT NOT NULL,
  reservation_date DATE NOT NULL,
  FOREIGN KEY (book_id) REFERENCES Books(book_id),
  FOREIGN KEY (member_id) REFERENCES Members(member_id)
);

CREATE TABLE Staff (
  staff_id INT PRIMARY KEY AUTO_INCREMENT,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  position VARCHAR(50),
  hire_date DATE
);

CREATE TABLE Departments (
  department_id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(50) NOT NULL,
  description TEXT
);

CREATE TABLE Staff_Departments (
  staff_id INT,
  department_id INT,
  PRIMARY KEY (staff_id, department_id),
  FOREIGN KEY (staff_id) REFERENCES Staff(staff_id),
  FOREIGN KEY (department_id) REFERENCES Departments(department_id)
);