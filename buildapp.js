const fs = require('fs');
const path = require('path');

const baseDir = './spring-boot-app';
const packageName = 'com.example.demo';
const packagePath = packageName.replace(/\./g, '/');
const javaDir = path.join(baseDir, `src/main/java/${packagePath}`);

// 1. Create Directory Structure
const dirs = [
    javaDir,
    path.join(javaDir, 'util'),
    path.join(javaDir, 'service'),
    path.join(javaDir, 'dto'),
    path.join(javaDir, 'entity'),
    path.join(javaDir, 'repository'),
    path.join(baseDir, 'src/main/resources')
];
dirs.forEach(dir => fs.mkdirSync(dir, { recursive: true }));

// 2. Simple POM.xml (H2 + ModelMapper)
const pomContent = `<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    <groupId>${packageName}</groupId>
    <artifactId>demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>org.modelmapper</groupId>
            <artifactId>modelmapper</artifactId>
            <version>2.4.4</version>
        </dependency>
    </dependencies>
</project>`;

// 3. Fixed application.yml (Forcing In-Memory H2)
// This overrides the "jdbc:postgresql" error by explicitly setting the H2 URL.
const ymlContent = `server:
  port: 8381
spring:
  datasource:
    url: jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1
    driver-class-name: org.h2.Driver
    username: sa
    password: password
  h2:
    console:
      enabled: true
      path: /h2-console
  jpa:
    database-platform: org.hibernate.dialect.H2Dialect
    hibernate:
      ddl-auto: update
    show-sql: true`;

// 4. Java Classes (Derived from source)
const utilClass = `package ${packageName}.util;
import org.modelmapper.ModelMapper;
import org.springframework.stereotype.Component;

@Component
public class ModelMapperUtil {
    private final ModelMapper modelMapper = new ModelMapper();
    public <D, E> D map(E source, Class<D> destinationClass) {
        return modelMapper.map(source, destinationClass);
    }
}`;

const serviceClass = `package ${packageName}.service;
import ${packageName}.util.ModelMapperUtil;
import ${packageName}.repository.UserRepository;
import ${packageName}.dto.UserDTO;
import ${packageName}.entity.User;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private final ModelMapperUtil modelMapperUtil;
    private final UserRepository userRepository;

    public UserService(ModelMapperUtil modelMapperUtil, UserRepository userRepository) {
        this.modelMapperUtil = modelMapperUtil;
        this.userRepository = userRepository;
    }

    public UserDTO createUser(UserDTO inputUserDto) {
        User user = modelMapperUtil.map(inputUserDto, User.class);
        user = userRepository.save(user);
        return modelMapperUtil.map(user, UserDTO.class);
    }
}`;

const entityClass = `package ${packageName}.entity;
import jakarta.persistence.*;

@Entity @Table(name="users")
public class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}`;

const dtoClass = `package ${packageName}.dto;
public class UserDTO {
    private Long id;
    private String name;
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}`;

const repoClass = `package ${packageName}.repository;
import ${packageName}.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {}`;

const mainClass = `package ${packageName};
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) { 
        SpringApplication.run(DemoApplication.class, args); 
    }
}`;

// Write all files
fs.writeFileSync(path.join(baseDir, 'pom.xml'), pomContent);
fs.writeFileSync(path.join(baseDir, 'src/main/resources/application.yml'), ymlContent);
fs.writeFileSync(path.join(javaDir, 'DemoApplication.java'), mainClass);
fs.writeFileSync(path.join(javaDir, 'util/ModelMapperUtil.java'), utilClass);
fs.writeFileSync(path.join(javaDir, 'service/UserService.java'), serviceClass);
fs.writeFileSync(path.join(javaDir, 'entity/User.java'), entityClass);
fs.writeFileSync(path.join(javaDir, 'dto/UserDTO.java'), dtoClass);
fs.writeFileSync(path.join(javaDir, 'repository/UserRepository.java'), repoClass);

console.log('✅ FIXED: H2 In-Memory App generated in ./spring-boot-app');