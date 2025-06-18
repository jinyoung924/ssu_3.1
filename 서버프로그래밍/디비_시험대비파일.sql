-- =================================================================
-- MySQL CRUD 기본 명령어 예제 파일
-- =================================================================
-- 이 스크립트는 'users'라는 테이블이 있다고 가정하고 작성되었습니다.
--
-- 테이블 구조 예시:
-- CREATE TABLE users (
--     id INT PRIMARY KEY AUTO_INCREMENT,
--     username VARCHAR(50) NOT NULL,
--     email VARCHAR(100)
-- );
-- =================================================================


-- 1. CREATE (생성): INSERT
-- 'users' 테이블에 새로운 데이터를 추가합니다.
INSERT INTO users (username, email) VALUES ('yuna', 'yuna@example.com');
INSERT INTO users (username, email) VALUES ('minjun', 'minjun@example.com');


-- 2. READ (조회): SELECT
-- 'users' 테이블의 데이터를 조회합니다.

-- (1) 모든 데이터 조회: 테이블의 모든 행과 컬럼을 가져옵니다.
SELECT * FROM users;

-- (2) 특정 조건의 데이터 조회: username이 'yuna'인 사용자만 조회합니다.
SELECT * FROM users WHERE username = 'yuna';


-- 3. UPDATE (수정): UPDATE
-- 'users' 테이블의 기존 데이터를 수정합니다.
-- ※※※ 주의: WHERE 절을 빠뜨리면 테이블의 모든 데이터가 변경될 수 있습니다. ※※※
UPDATE users SET email = 'yuna.kim@example.com' WHERE id = 1;


-- 확인을 위해 수정된 데이터를 다시 조회합니다.
SELECT * FROM users WHERE id = 1;


-- 4. DELETE (삭제): DELETE
-- 'users' 테이블에서 데이터를 삭제합니다.
-- ※※※ 주의: WHERE 절을 빠뜨리면 테이블의 모든 데이터가 삭제될 수 있습니다. ※※※
DELETE FROM users WHERE username = 'minjun';


-- 최종적으로 남은 데이터를 확인합니다.
SELECT * FROM users;