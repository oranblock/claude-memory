// server.go
package main

import (
	"log"
	"net/http"
	
	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/api/users", GetUsers).Methods("GET")
	r.HandleFunc("/api/users/{id}", GetUser).Methods("GET")
	
	log.Fatal(http.ListenAndServe(":8080", r))
}

func GetUsers(w http.ResponseWriter, r *http.Request) {
	// Implementation here
}