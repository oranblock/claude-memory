import mongoose from 'mongoose';
import { TaskService } from '../taskService';
import { Task } from '../../models/Task';
import { Category } from '../../models/Category';
import { connectDatabase, closeDatabaseConnection } from '../../config/database';

// Mock data
const userId = new mongoose.Types.ObjectId();
const categoryId = new mongoose.Types.ObjectId();
const taskId = new mongoose.Types.ObjectId();

const mockTask = {
  _id: taskId,
  title: 'Test Task',
  description: 'Test Description',
  status: 'not-started',
  priority: 'medium',
  userId,
  categoryId,
  createdAt: new Date(),
  updatedAt: new Date(),
};

// Mock Task model
jest.mock('../../models/Task', () => ({
  Task: {
    create: jest.fn(),
    findById: jest.fn(),
    findOne: jest.fn(),
    find: jest.fn(),
    findByIdAndUpdate: jest.fn(),
    findByIdAndDelete: jest.fn(),
  },
}));

describe('TaskService', () => {
  let taskService: TaskService;

  beforeAll(async () => {
    taskService = new TaskService();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a new task', async () => {
      (Task.create as jest.Mock).mockResolvedValueOnce(mockTask);

      const taskData = {
        title: 'Test Task',
        description: 'Test Description',
        status: 'not-started' as const,
        priority: 'medium' as const,
        userId: userId.toString(),
        categoryId: categoryId.toString(),
      };

      const result = await taskService.create(taskData);

      expect(Task.create).toHaveBeenCalledWith(expect.objectContaining({
        title: 'Test Task',
        description: 'Test Description',
        status: 'not-started',
        priority: 'medium',
      }));
      expect(result).toEqual(mockTask);
    });
  });

  describe('findById', () => {
    it('should find a task by ID', async () => {
      (Task.findById as jest.Mock).mockReturnValueOnce({
        populate: jest.fn().mockResolvedValueOnce(mockTask),
      });

      const result = await taskService.findById(taskId.toString());

      expect(Task.findById).toHaveBeenCalledWith(taskId);
      expect(result).toEqual(mockTask);
    });

    it('should return null if task not found', async () => {
      (Task.findById as jest.Mock).mockReturnValueOnce({
        populate: jest.fn().mockResolvedValueOnce(null),
      });

      const result = await taskService.findById('non-existent-id');

      expect(result).toBeNull();
    });
  });

  describe('findByIdAndUserId', () => {
    it('should find a task by ID and user ID', async () => {
      (Task.findOne as jest.Mock).mockReturnValueOnce({
        populate: jest.fn().mockResolvedValueOnce(mockTask),
      });

      const result = await taskService.findByIdAndUserId(
        taskId.toString(),
        userId.toString()
      );

      expect(Task.findOne).toHaveBeenCalledWith({
        _id: taskId,
        userId,
      });
      expect(result).toEqual(mockTask);
    });
  });

  describe('findAllByUserId', () => {
    it('should find all tasks for a user', async () => {
      (Task.find as jest.Mock).mockReturnValueOnce({
        populate: jest.fn().mockReturnValueOnce({
          sort: jest.fn().mockResolvedValueOnce([mockTask]),
        }),
      });

      const result = await taskService.findAllByUserId(userId.toString());

      expect(Task.find).toHaveBeenCalledWith({ userId });
      expect(result).toEqual([mockTask]);
    });
  });

  describe('update', () => {
    it('should update a task', async () => {
      const updatedTask = { ...mockTask, title: 'Updated Title' };
      (Task.findByIdAndUpdate as jest.Mock).mockReturnValueOnce({
        populate: jest.fn().mockResolvedValueOnce(updatedTask),
      });

      const result = await taskService.update(
        taskId.toString(),
        { title: 'Updated Title' }
      );

      expect(Task.findByIdAndUpdate).toHaveBeenCalledWith(
        taskId,
        expect.objectContaining({ title: 'Updated Title' }),
        { new: true }
      );
      expect(result).toEqual(updatedTask);
    });
  });

  describe('delete', () => {
    it('should delete a task', async () => {
      (Task.findByIdAndDelete as jest.Mock).mockResolvedValueOnce(mockTask);

      await taskService.delete(taskId.toString());

      expect(Task.findByIdAndDelete).toHaveBeenCalledWith(taskId);
    });
  });
});